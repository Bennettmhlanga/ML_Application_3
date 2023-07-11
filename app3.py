import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
st.set_option('deprecation.showPyplotGlobalUse', False)

from PIL import Image

st.title('CLASSIFICATION APP 2')
image = Image.open('mother_nature.jpg')
st.image(image,use_column_width=True)

def main():
	activities = ['EDA','Visualization','Dashboard','Model','About us']
	option=st.sidebar.selectbox('Select option',activities)

	#EDA SECTION
	if option=='EDA':
		st.subheader('Lets perform some exploratory data analysis')
		data=st.file_uploader('Add dataset',type=['csv','xlsx','txt','json'])
		st.success('Dataset loaded')
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Display shape'):
				st.write(df.shape)

			if st.checkbox('Display columns'):
				st.write(df.columns)

			if st.checkbox('Check null values'):
				st.write(df.isna().sum())

			if st.checkbox('Check data types'):
				st.write(df.dtypes)

			if st.checkbox('Display correlation'):
				st.write(df.corr())

			if st.checkbox('Check data summary'):
				st.write(df.describe().T)

			if st.checkbox('Select multiple columns'):
				selected_columns=st.multiselect('Select the columns you want', df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

	elif option=='Visualization':
		st.subheader('Lets visualize our data')
		data = st.file_uploader('Add a dataset', type=['csv','xlsx','txt','json'])
		st.success('Dataset loaded')

		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Select columns to plot'):
				selected_columns=st.multiselect('Select columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

			if st.checkbox('Display the correlation heatmap'):
				st.write(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
				st.pyplot()

			if st.checkbox('Display Pairplot'):
				st.write(sns.pairplot(df,diag_kind='kde'))
				st.pyplot()
			
			if st.checkbox('Display Box plot'):
				columns=st.multiselect('Select columns',df.columns)
				st.write(sns.boxplot(columns))
				st.pyplot()

			if st.checkbox('Display distribution plot'):
				columns=st.multiselect('Select columns',df.columns)
				st.write(sns.distplot(columns))
				st.pyplot()

			if st.checkbox('Display Pie Chart'):
				all_columns=df.columns.to_list()
				pie_columns=st.selectbox("select column to display",all_columns)
				pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pieChart)
				st.pyplot()

			if st.checkbox('Display histogram'):
				columns=st.multiselect('Select columns',df.columns)
				histogram=df[columns].hist(stacked=False, bins =100, figsize = (12,30), layout=(14,2));
				st.write(histogram)
				st.pyplot()

	elif option=='Model':
		st.subheader('Lets build the model')
		data=st.file_uploader('Add dataset',type=['csv','xlsx','txt','json'])
		st.success('Dataset loaded')
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Select columns'):
				new_data=st.multiselect('Select your columns and the target/dependent variable should be selected last',df.columns)
				df1=df[new_data]
				st.dataframe(df1)

				X=df1.iloc[:,0:-1]
				y=df1.iloc[:,-1]
			seed =st.sidebar.slider('Seed',1,200)
			classifier_name=st.sidebar.selectbox("Select a classifier",('KNN','LR','Naive bayes','Decision tree'))#We removed svm

			def add_parameter(name_of_classifier):
				params=dict()
				if name_of_classifier=='SVM':
					C=st.sidebar.slider('C',0.01,15.0)
					params['C']=C
				else:
					name_of_classifier=='KNN'
					K=st.sidebar.slider('K',1,15)
					params['K']=K
					return params

			params=add_parameter(classifier_name)

			def get_classifier(name_of_classifier,params):
				clf=None
				if name_of_classifier=='SVM':
					clf=SVC(C=params['C'])
				elif name_of_classifier=='KNN':
					clf=KNeighborsClassifier(n_neighbors=params['K'])
				elif name_of_classifier=='LR':
					clf=LogisticRegression()
				elif name_of_classifier=='Naive bayes':
					clf=GaussianNB()
				elif name_of_classifier=='Decision tree':
					clf=DecisionTreeClassifier()
				else:
					st.warning('Choose an algorithm to use')
				return clf

			clf=get_classifier(classifier_name, params)
			X=df.iloc[:,0:-1]
			y=df.iloc[:,-1]

			X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2,random_state=seed)
			clf.fit(X_train,y_train)
			y_pred=clf.predict(X_test)
			st.write("Prediction", y_pred)
			accuracy=accuracy_score(y_test,y_pred)
			st.write('Classifier:',classifier_name)
			st.write('Accuracy', accuracy)
	elif option=='About us':
		st.markdown('This interactive ML project was developed by Bennett Mhlanga under the supervision of Dr Briit. If you have any queries kindly reach out on bennettmhlanga959@gmail.com we will get back to you. I kindly ask you to add a dataset that has already been cleaned. This work is nothing more but a demonstration of how we can present our work to stackeholders who are not interested in the code part.')


if __name__ =='__main__':
	main()


